class RecommendationService
  RECOMMENDATION_SERVICE_URL = ENV.fetch('RECOMMENDATION_SERVICE_URL', 'http://localhost:8000')

  def recommend_for_user(user_id, limit = 5)
    return [] unless user_id.present?
    
    response = Faraday.get("#{RECOMMENDATION_SERVICE_URL}/recommend/user/#{user_id}?limit=#{limit}")
    JSON.parse(response.body)['recommendations'] rescue []
  end

  def recommend_for_items(item_ids, limit = 5)
    return [] if item_ids.blank?
    
    response = Faraday.get(
      "#{RECOMMENDATION_SERVICE_URL}/recommend/items/?item_ids=#{item_ids.join(',')}&limit=#{limit}"
    )
    JSON.parse(response.body)['recommendations'] rescue []
  end
end